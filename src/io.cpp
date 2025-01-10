#include "io.h"


void ParseEquals(const std::string &line, std::string &lhs,
                                std::string &rhs)
{
    /// Pull out lhs and rhs and eliminate any spaces.
    size_t beg = line.find_first_not_of(" ");
    size_t end = line.find_first_of("=");
    // Check for no parameter name
    if (beg == end)
        throw 1;
    // Check for no parameter value
    if (end != line.find_last_of("="))
        throw 1;
    // Check for no equals sign
    if (end == std::string::npos)
        throw 1;

    lhs = line.substr(line.find_first_not_of(" "), end - beg);
    lhs = lhs.substr(0, lhs.find_last_not_of(" ") + 1);
    rhs = line.substr(line.find_last_of("=") + 1);
    rhs = rhs.substr(rhs.find_first_not_of(" "));
    rhs = rhs.substr(0, rhs.find_last_not_of(" ") + 1);
}


Inputs* ReadXmlFile(const char* filename)
{
    TiXmlDocument *m_xmlDoc = new TiXmlDocument;
    TiXmlDocument doc( filename );
    Inputs* inp = new Inputs;
    doc.LoadFile();
    
    TiXmlHandle hDoc(&doc);
    
//    TiXmlHandle docHandle(m_xmlDoc);
    
//    TiXmlElement *e;
//
//    e = doc->FirstChildElement("METRIC").Element();
//
//    TiXmlElement *parametersElement =
//        conditions->FirstChildElement("PARAMETERS");
    
    TiXmlElement *xmlMetric = doc.FirstChildElement("DGSOLVER");
    
    
    TiXmlElement *xmlParam = xmlMetric->FirstChildElement("PARAMETERS");
    
    std::map<std::string,std::string> param_map;
    if (xmlParam)
    {
        TiXmlElement *parameter = xmlParam->FirstChildElement("P");
        
        while (parameter)
        {
            TiXmlNode *node = parameter->FirstChild();
            
            std::string line = node->ToText()->Value(), lhs, rhs;
            
            try
            {
                ParseEquals(line, lhs, rhs);
            }
            catch (...)
            {
                std::cout << "Error reading metric.xml " << std::endl;
            }
            
            if (!lhs.empty() && !rhs.empty())
            {
                // double value = std::stod(rhs);
                param_map[lhs] = rhs;
                
            }
            parameter = parameter->NextSiblingElement();
        }
    }
    
    if(param_map.find("PolynomialOrder")!=param_map.end())
    {
        inp->porder = std::stod(param_map["PolynomialOrder"]);
    }
    else
    {
        std::cout << "Error: PolynomialOrder is not defined in inputs.xml." << std::endl;
    }
    if(param_map.find("nElements")!=param_map.end())
    {
        inp->nelem = std::stod(param_map["nElements"]);
    }
    else
    {
        std::cout << "Error: nElements is not defined in inputs.xml." << std::endl;
    }
    if(param_map.find("dt")!=param_map.end())
    {
        inp->dt = std::stod(param_map["dt"]);
    }
    else
    {
        std::cout << "Error: dt is not defined in inputs.xml." << std::endl;
    }
    if(param_map.find("nt")!=param_map.end())
    {
        inp->nt = std::stod(param_map["nt"]);
    }
    else
    {
        std::cout << "Error: nt is not defined in inputs.xml." << std::endl;
    }
    if(param_map.find("BasisType")!=param_map.end())
    {
        inp->btype = param_map["BasisType"].c_str();
    }
    else
    {
        std::cout << "Error: BasisType is not defined in inputs.xml." << std::endl;
    }
    if(param_map.find("PointsType")!=param_map.end())
    {
        inp->ptype = param_map["PointsType"].c_str();
    }
    else
    {
        std::cout << "Error: PointsType is not defined in inputs.xml." << std::endl;
    }
    if(param_map.find("nQuadrature")!=param_map.end())
    {
        inp->nquad = std::stod(param_map["nQuadrature"]);
    }
    else
    {
        std::cout << "Error: nQuadrature is not defined in inputs.xml." << std::endl;
    }
    if(param_map.find("TimeScheme")!=param_map.end())
    {
        // inp->timescheme = std::stod(param_map["TimeScheme"]);
        inp->timescheme = param_map["TimeScheme"].c_str();
    }
    else
    {
        std::cout << "Error: TimeScheme is not defined in inputs.xml." << std::endl;
    }
    if(param_map.find("Restart")!=param_map.end())
    {
        inp->restart = std::stod(param_map["Restart"]);
    }
    else
    {
        std::cout << "Error: nQuadrature is not defined in inputs.xml." << std::endl;
    }
    
    
        std::cout << "===================================================" << std::endl;
        std::cout << "============== 1D DG Solver Inputs ================" << std::endl;
        std::cout << "===================================================" << std::endl;
        std::cout << "porder        = " << inp->porder << std::endl;
        std::cout << "nelem         = " << inp->nelem << std::endl;
        std::cout << "dt            = " << inp->dt << std::endl;
        std::cout << "nt            = " << inp->nt << std::endl;
        std::cout << "BasisType     = " << inp->btype << std::endl;
        std::cout << "PointsType    = " << inp->ptype << std::endl;
        std::cout << "nQuadrature   = " << inp->nquad << std::endl;
        std::cout << "TimeScheme    = " << inp->timescheme << std::endl;
        std::cout << "===================================================" << std::endl;
        std::cout << "===================================================" << std::endl;
        std::cout << "===================================================" << std::endl;
        std::cout << "  " << std::endl;
    




    return inp;
}